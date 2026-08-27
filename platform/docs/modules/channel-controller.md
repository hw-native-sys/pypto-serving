# Channel Controller Module

Source path: `include/modules/channelController/module.hpp`

The channel controller manages channel creation through a desired-state reconciliation loop. It turns desired producer and consumer registrations into initialized HiCR-backed `Output` and `Input` channels.

## Main Type

`serving::modules::channelController::Module` derives from `serving::modules::Module` and runs as a periodic service by default.

Constructor inputs:

- Current instance id.
- Ordered list of `HiCR::CommunicationManager *` objects.
- Global memory-slot exchange tag.
- Reconciliation interval in milliseconds.

At least one communication manager is required.

## Desired State

The controller keeps separate desired and actual maps for producers and consumers.

`addDesiredProducer(targetId, channelId, edge, keyBuilder)` creates a desired `Output` channel for an edge.

`addDesiredConsumer(sourceId, channelId, edge, keyBuilder)` creates a desired `Input` channel for an edge.

`removeDesiredProducer(edgeName)` and `removeDesiredConsumer(edgeName)` remove desired entries.

`hasProducer(edgeName)` and `hasConsumer(edgeName)` report whether actual channels exist.

`getProducer(edgeName)` and `getConsumer(edgeName)` return initialized actual channels.

## Reconciliation Flow

The private `reconcile()` method:

1. Diffs desired producers/consumers against actual producers/consumers.
2. Collects memory slots required by missing channels.
3. Groups slots by communication manager.
4. Exchanges global memory slots through the configured managers.
5. Fences each manager on the exchange tag.
6. Initializes missing `Output` and `Input` channels.
7. Moves initialized channels into actual maps.
8. Removes stale actual channels when desired entries disappear.

The module calls `reconcile()` during `initialize()` and from its periodic `service()` method.

## Relation To Platform Design

This module implements the channel-management direction from issue #32: model support can request channels by desired intent, while platform code handles lifecycle and resource exchange.

The current implementation works at the HiCR channel level. Future integration can add a higher-level distinction between host control channels and device tensor payload channels while preserving the same desired-state reconciliation model.

## Example

`examples/modules/channelController/channelController.cpp` parses a deployment, assigns edge managers, registers desired local producer/consumer channels, waits until both are ready, runs a telephone-game message exchange, removes desired channels, and terminates.

## Current Limitations

- The TODO in the source notes that every instance currently needs to run this service for MPI-style backends.
- The controller does not yet expose a frontend that hides backend-specific details from callers.
- Placement policy is provided indirectly through edge HiCR manager assignments.
