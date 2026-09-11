# Broadcast Deployment Module

Source path: `include/modules/broadcastDeployment/module.hpp`

The broadcast deployment module distributes a serialized deployment configuration from a deployer instance to other instances through RPC.

## Main Type

`serving::modules::broadcastDeployment::Module` derives from `serving::modules::Module`.

Constructor inputs:

- `HiCR::InstanceManager`
- `HiCR::ComputeManager`
- `HiCR::frontend::RPCEngine`
- Deployer instance id.
- Current instance id.
- Optional `serving::configuration::Deployment` for the deployer.

The deployer constructor receives the deployment object. Worker instances use the constructor without a deployment and fetch it during initialization.

## RPC Flow

All instances register an RPC target named `__SERVING_REQUEST_DEPLOYMENT_CONFIGURATION_RPC_NAME`.

When a worker initializes:

1. It requests the deployment RPC from the deployer instance.
2. It waits for the return value.
3. It parses the returned JSON string.
4. It constructs a local `configuration::Deployment` from that JSON.
5. It frees the RPC return-value memory slot.

When the deployer initializes:

- It listens for deployment requests from non-deployer instances.
- On request, it serializes `_deployment` to JSON and submits the serialized string as the RPC return value.

## Relation To Platform Design

This module implements the static bootstrap direction from issue #32. It ensures all instances can obtain the same desired deployment state before channels, services, coordinators, and replicas are initialized.

## Example

`examples/modules/broadcastDeployment/broadcastDeployment.cpp` parses a deployment only on the root instance, creates the broadcast module on every instance, initializes the engine, and prints the received deployment on each instance.

## Current Limitations

- Deployment distribution is init-only and does not publish incremental updates.
- There is no versioning or consistency protocol for dynamic reconfiguration.
- The deployer listens in a loop based on the current instance list and assumes startup-time distribution.
