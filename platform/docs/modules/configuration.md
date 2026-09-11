# Configuration Module

Source paths:

- `include/modules/configuration/deployment.hpp`
- `include/modules/configuration/partition.hpp`
- `include/modules/configuration/task.hpp`
- `include/modules/configuration/replica.hpp`
- `include/modules/configuration/edge.hpp`
- `include/modules/configuration/requestManager.hpp`

The configuration module represents the desired deployment graph for the platform layer. It describes partitions, host-side tasks, replicas, request-manager endpoints, and communication edges.

## Main Types

`serving::configuration::Deployment` is the top-level object. It owns:

- A deployment name.
- A list of `Partition` objects.
- A list of `Edge` objects.
- A `RequestManager` object.
- Heartbeat settings.
- Control-buffer settings.

`Partition` describes a logical stage of the application. It stores:

- A partition name.
- The coordinator instance id.
- A list of `Task` objects.
- A list of `Replica` objects.

`Task` describes a named function within a partition. It stores:

- Function name.
- Input edge names.
- Output edge names.
- Intra-partition dependencies by function name.

`Replica` identifies a concrete instance assigned to execute a partition.

`Edge` describes a communication link. It stores:

- Edge name.
- Producer partition.
- Consumer partition.
- Buffer capacity and size.
- Payload HiCR manager objects.
- Coordination HiCR manager objects.
- Prompt/result flags for request-manager boundary edges.

`RequestManager` defines the external entry and result edges for the deployment.

## Serialization

All configuration objects serialize to and deserialize from `nlohmann::json`. Runtime-only HiCR manager pointers and memory spaces are not serialized; they are assigned after parsing by runtime setup code.

The serialized deployment shape includes:

- `Name`
- `Partitions`
- `Edges`
- `Request Manager`
- `Settings`

## Validation

`Deployment::verify()` performs graph sanity checks and fills edge producer/consumer metadata.

Important checks include:

- Task dependencies must refer to tasks in the same partition.
- Tasks cannot depend on themselves.
- Edge names must not be duplicated.
- Inputs and outputs must refer to defined edges.
- Each non-request-manager edge must have exactly one producer and one consumer.
- An edge cannot connect a partition to itself, except request-manager prompt/result edges.
- The request-manager input and output edges must be used.
- The partition consuming the external input must not also consume cross-partition inputs.
- The partition producing the external output must not also produce cross-partition outputs.

After validation, request-manager boundary edges are marked as prompt or result edges.

## Relation To Platform Design

The deployment graph is the current implementation of the static deployment API described in issue #32. It gives the platform a desired-state view of partitions, tasks, replicas, and edges before dynamic scaling or fault recovery are added.

## Current Limitations

- Placement concepts such as host/device placement are not explicit JSON fields yet.
- Replicas and coordinator instance ids are optional because they may be assigned at runtime, but no placement optimizer is implemented here.
- Heartbeat settings are represented, but recovery policy is not implemented in this module.
- Edge payload/control placement is represented through runtime HiCR manager assignments rather than a high-level placement enum.
