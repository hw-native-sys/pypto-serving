# Model Support Quick-Start

This guide shows how to wire a model's inference kernel into the platform coordinator→replica path, using the `singlePartition` benchmark as the reference example.

## 1. The `processFc` interface

The platform invokes your inference code through a single callback:

```cpp
using processFc_t = std::function<void(serving::modules::roles::TaskContext &)>;
```

Inside the callback:

```cpp
void myInferenceFn(serving::modules::roles::TaskContext &ctx)
{
  // Read input tensor
  auto slot = ctx.getInput("tokens");   // returns shared_ptr<HiCR::LocalMemorySlot>
  const auto *data = static_cast<const uint8_t *>(slot->getPointer());
  size_t      size = slot->getSize();

  // Run inference ...
  std::vector<float> logits = runModel(data, size);

  // Write output (platform takes a copy; the buffer can go out of scope after this call)
  ctx.setOutput("logits", logits.data(), logits.size() * sizeof(float));
}
```

`TaskContext` API summary:

| Method | Description |
|--------|-------------|
| `getInput(name)` | Returns the named input slot; `nullptr` if the dependency was not satisfied |
| `setOutput(name, ptr, size)` | Copies `size` bytes from `ptr` into the named output channel |
| `setOutput(name, slot)` | Moves an already-allocated `LocalMemorySlot` into the output (zero-copy path) |

## 2. Defining the data graph

Each edge in the deployment graph carries one tensor per job. Configure capacity (concurrent jobs in flight) and payload size to match your model's I/O shapes:

```cpp
// One input edge: 4096-byte token buffer, up to 4 jobs in flight
auto tokensEdge = std::make_shared<serving::configuration::Edge>(
    "tokens", /*capacity=*/4, /*payloadBytes=*/4096);
tokensEdge->setPayloadCommunicationManager(commMgr);
tokensEdge->setPayloadMemoryManager(memMgr);
tokensEdge->setPayloadMemorySpace(memSpace);
tokensEdge->setCoordinationCommunicationManager(commMgr);
tokensEdge->setCoordinationMemoryManager(memMgr);
tokensEdge->setCoordinationMemorySpace(memSpace);

// One output edge: 8-byte result (uint64_t logit checksum in the example)
auto logitsEdge = std::make_shared<serving::configuration::Edge>(
    "logits", /*capacity=*/4, sizeof(uint64_t));
// ... same manager/space setters ...

serving::configuration::Deployment deployment;
deployment.addChannel(tokensEdge);
deployment.addChannel(logitsEdge);
```

Add one task describing the I/O contract:

```cpp
auto task = std::make_shared<serving::configuration::Task>(std::string("infer"));
task->addInput("tokens");
task->addOutput("logits");

auto partition = std::make_shared<serving::configuration::Partition>("P0", instanceId);
partition->addTask(task);
partition->addReplica(std::make_shared<serving::configuration::Replica>(instanceId));
deployment.addPartition(partition);
```

## 3. Wiring coordinator and replica (co-located)

For a single-rank deployment the coordinator and replica live on the same MPI rank. They communicate over loopback MPI channels. The internal channel IDs (`1000`, `2000` below) must not conflict with any other channels in the system.

```cpp
// Internal edges derived from the graph edge templates
const auto internalTokensEdge = makeInternalEdgeFromTemplate(
    "tokens-coord-replica", *deployment.getEdges()[0]);
const auto internalLogitsEdge = makeInternalEdgeFromTemplate(
    "logits-replica-coord", *deployment.getEdges()[1]);

// Coordinator side: sends tokens, receives logits
auto tokensOut = channelController->addDesiredProducer(
    instanceId, /*channelId=*/1000, internalTokensEdge, defaultChannelKeyBuilder).lock();
auto logitsIn  = channelController->addDesiredConsumer(
    instanceId, /*channelId=*/2000, internalLogitsEdge, defaultChannelKeyBuilder).lock();

coordinator->addReplica(instanceId,
    /*inputs=*/ {{"tokens", tokensOut}},
    /*outputs=*/{{"logits",  logitsIn}});

// Replica side: receives tokens, sends logits
auto tokensIn  = channelController->addDesiredConsumer(
    instanceId, /*channelId=*/1000, internalTokensEdge, defaultChannelKeyBuilder).lock();
auto logitsOut = channelController->addDesiredProducer(
    instanceId, /*channelId=*/2000, internalLogitsEdge, defaultChannelKeyBuilder).lock();

replica->setCoordinator(instanceId,
    /*outputs=*/{{"logits",  logitsOut}},
    /*inputs=*/ {{"tokens",  tokensIn}});
```

## 4. Submitting jobs

After `serving.initialize()` and `serving.run()`, jobs are submitted directly to the coordinator. The platform routes each job through the channel, invokes `processFc`, and delivers the output via the completion callback.

```cpp
// Register a completion callback before submitting
coordinator->setCompletionCallback(
    [](const serving::system::channels::Message::metadata_t &meta,
       const std::vector<serving::modules::roles::coordinator::Module::JobOutput> &outputs)
    {
      // outputs[i].data is a LocalMemorySlot containing the named output
      auto *result = static_cast<const float *>(outputs[0].data->getPointer());
      // ... process result ...
    });

// Submit a job
auto job = std::make_shared<serving::modules::roles::Job>(
    jobFactory.createJob("infer", metadata));
job->getInputDependency("tokens").storeData(inputBuffer.data(), inputBuffer.size());
job->getInputDependency("tokens").setSatisfied(true);
coordinator->submitJob(job);
```

`metadata.sequenceId` is returned verbatim in the completion callback and can be used to correlate requests with responses.

## 5. Thread safety and memory ownership

- `coordinator->submitJob()` is thread-safe; call it from any thread.
- `storeData()` copies the buffer into platform-managed memory; the caller's buffer can be reused immediately after `storeData` returns.
- The `LocalMemorySlot` passed to the completion callback is valid only for the duration of the callback. Copy the data out before returning.
- `processFc` is invoked by the replica's internal thread. Do not share mutable state between `processFc` and the submission thread without synchronization.

## 6. Performance characteristics

The platform adds one round-trip through the MPI loopback channel per job. With a 1 ms `channelDispatcher` poll interval the overhead is ~1 ms/job independent of payload size.

| Compute time | Platform overhead | Assessment |
|:---:|:---:|:---:|
| < 1 ms | > 100% | Framework dominates — optimize poll interval or batch |
| ~10 ms | ~10% | Borderline — acceptable for many workloads |
| ≥ 100 ms | < 1% | Not in critical path |

Run the `singlePartition` benchmark to measure overhead on your hardware:

```sh
mpirun -np 1 --oversubscribe singlePartition <num_requests> <compute_us>

# Example: 200 requests, 10 ms simulated compute
mpirun -np 1 --oversubscribe singlePartition 200 10000
```

Output:
```
[Baseline]  200 calls | 2000.235 ms total | 10.001 ms/call
[Platform]  200 jobs  | 2205.507 ms total | 11.028 ms/call
[Overhead]  +1.026 ms/call (10.3% of compute time)
            compute=10000 us/call → framework borderline critical path
```

To reduce overhead below 1 ms, lower the `channelDispatcher` poll interval or reduce payload copy volume via the zero-copy `setOutput(name, slot)` path.
